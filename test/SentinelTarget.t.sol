// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/SentinelTarget.sol";

contract SentinelTargetTest is Test {
    SentinelTarget target;
    address user;

    function setUp() public {
        target = new SentinelTarget();
        user = makeAddr("user");
        vm.deal(user, 10 ether);
    }

    function testSetValue() public {
        target.setValue(42);
        assertEq(target.value(), 42);
    }

    function testDeposit() public {
        vm.prank(user);
        target.deposit{value: 1 ether}();

        assertEq(target.balances(user), 1 ether);
        assertEq(address(target).balance, 1 ether);
    }

    function testWithdraw() public {
        vm.startPrank(user);
        target.deposit{value: 1 ether}();
        target.withdraw(0.4 ether);
        vm.stopPrank();

        assertEq(target.balances(user), 0.6 ether);
        assertEq(address(target).balance, 0.6 ether);
    }

    function testCannotWithdrawMoreThanBalance() public {
        vm.prank(user);
        target.deposit{value: 1 ether}();

        vm.prank(user);
        vm.expectRevert("Insufficient balance");
        target.withdraw(2 ether);
    }
}
